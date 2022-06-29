# **Sebastian Ramos and Tomas Vera Audio Feature Extraction Project**


In this project, we wrote a Python program that analyzes the features of audio recordings from different musical genres: rock, classical, jazz, and speech.


Part 1

For 2 examples from each genre, we computed and plotted the audio signal waveform as well as the magnitude spectrum of the audio signal.


Part 2 

Next, we analyzed the signals to obtain summary statistics that highlight differences between the music genres using block-based analysis. We use block_based analysis to computer extract short-time (local) features, and then summarize these local short-time features with a summary statistic (mean, standard deviation)


Project datafiles provided by Yon Visell


```python
import numpy as np
import matplotlib.pyplot as plt
import essentia.standard as es
import IPython as ipy
```

# **Part 1**
Plot audio signal waveform as well as the magnitude spectrum of the audio signal


```python
import glob

#load files into dictionary with key value corresponding to genre
mediaDir = './music_dataFolder'
files = {}
files['classical'] = glob.glob(mediaDir + '/classical/'+'*.wav')
files['jazz'] = glob.glob(mediaDir + '/jazz/'+'*.wav')
files['rockblues'] = glob.glob(mediaDir + '/rockblues/'+'*.wav')
files['speech'] = glob.glob(mediaDir + '/speech/'+'*.wav')

Fs = 44100
Ts = 1/Fs 

#print 2 files from each genre
for k, v in files.items():
  
  for i in range(2):
    #print basic information of audio file
    print()
    print("-----" + k + " file " + str(i+1) + "-----")
    print(v[i])
    
    #load current audio recording through essentia
    song = es.MonoLoader(filename=v[i], sampleRate=Fs)()

    #get the number of samples in the current audio recording
    n_samples = song.shape[0]

    #get the amplitude domain value for the each sample
    ff = np.arange(0,int(Fs),Fs/n_samples)

    #get the time domain value for each sample
    tt = np.arange(0,n_samples*Ts,Ts)

    #calculate the fast fourier transform for the currect recording
    Sk = np.fft.fft(song)
    ASk = np.abs(Sk) / n_samples

    #plot the audio signal waveform for the current audio file
    plt.figure(figsize=(14, 4))
    plt.subplot(1, 2, 1)
    plt.title("Audio Signal Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.plot(tt, song[:len(tt)])

    #plot the spectrum chart for the current audio file
    plt.subplot(1, 2, 2)
    plt.plot(ff[0:int(n_samples/5)], ASk[0:int(n_samples/5)])
    plt.title("Magnitude Spectrum")
    plt.ylabel("Amplitude")
    plt.xlabel("Freq, Hz")  
    plt.show()
```

    
    -----classical file 1-----
    ./music_dataFolder/classical/classical1.wav



    
![png](signal_feature_extraction_project_files/signal_feature_extraction_project_4_1.png)
    


    
    -----classical file 2-----
    ./music_dataFolder/classical/classical2.wav



    
![png](signal_feature_extraction_project_files/signal_feature_extraction_project_4_3.png)
    


    
    -----jazz file 1-----
    ./music_dataFolder/jazz/ipanema.wav



    
![png](signal_feature_extraction_project_files/signal_feature_extraction_project_4_5.png)
    


    
    -----jazz file 2-----
    ./music_dataFolder/jazz/duke.wav



    
![png](signal_feature_extraction_project_files/signal_feature_extraction_project_4_7.png)
    


    
    -----rockblues file 1-----
    ./music_dataFolder/rockblues/rock2.wav



    
![png](signal_feature_extraction_project_files/signal_feature_extraction_project_4_9.png)
    


    
    -----rockblues file 2-----
    ./music_dataFolder/rockblues/hendrix.wav



    
![png](signal_feature_extraction_project_files/signal_feature_extraction_project_4_11.png)
    


    
    -----speech file 1-----
    ./music_dataFolder/speech/teachers1.wav



    
![png](signal_feature_extraction_project_files/signal_feature_extraction_project_4_13.png)
    


    
    -----speech file 2-----
    ./music_dataFolder/speech/dialogue1.wav



    
![png](signal_feature_extraction_project_files/signal_feature_extraction_project_4_15.png)
    


On initial inspection, it is apparent that some genres contain more distinguishing traits than others. Based on the two samples extracted from each genre in the dataset, it appears that classical music contains the majority of its frequency content evenly distributed between 0 and 1k Hz. Jazz shares a similar characteristic within a broader ranging from 2kHz to 3kHz. Rockblues demonstrates concentrated frequency components of high magnitude between 0 and 2k Hz. Finally, the speech samples show a more dense spectrum, suggesting the frequency components share similar magnitude and prominent peaks are less frequent.

# **Part 2**

## Feature 1: Mean Spectral Centroid

The magnitude spectrum variation of the spectral centroid is calculated in order to determine the average frequency in which the weighted spectrum resides.
The average of the spectral centroids for each block is determined for each audio file as shown:

$v_{SC,block}(i)=\frac{\sum_{k=0}^{block\_size-1} k|X(i,k)|}{\sum_{k=0}^{block\_size-1} |X(i, k)|}$

<br>

$v_{SC,mean}=\frac{\sum_{i=0}^{n\_blocks-1} v_{SC,block}(i)}{n\_blocks}$


```python
#define dictionary for calculated mean spectral centroid values for each genre
mean_spectral_centroids = {
    'classical': [],
    'jazz': [],
    'rockblues': [],
    'speech': []
}

#loop through dictionary of audio recording genres
for k, v in files.items():

  #loop through each recording for the corresponding genre
  for i in range(len(v)):

    #print basic information of audio file
    print()
    print("-----" + k + " file " + str(i+1) + "-----")
    print(v[i])
    
    #load audio recording through essentia
    song = es.MonoLoader(filename=v[i], sampleRate=Fs)()

    #get number of samples for the current audio recording
    n_samples = song.shape[0]

    #define basic parameters for block based analysis of recording
    block_size = 100000 
    overlap = 0.25  #25% overlap between blocks
    block_step = (1-overlap)*block_size  
    n_blocks = np.ceil(n_samples/block_step)

    #initialize zero np array to store spectral centroid for each block
    centroid_list = np.zeros(int(n_blocks))

    #loop through all predefined blocks of audio recording
    for i in range(int(n_blocks)):

      #calculate start and end sample step for current block
      block_s = int(i*block_step)
      block_e = int(min(block_s + block_size, n_samples))

      #calculate audio data for the specific block
      block = song[block_s:block_e]

      #define number of samples for the current block
      Nsamples = block.shape[0]

      ff = np.arange(0,int(Fs),Fs/Nsamples)

      #calculate the fast fourier transform for the current block
      block_fft = np.abs(np.fft.fft(block))

      #only display half of the samples of the audio file for better FFT graph plot
      ff = ff[:int(Nsamples/2)]
      block_fft = block_fft[:int(Nsamples/2)]

      #calculate current block's spectral centroid
      centroid_list[i] = np.sum(ff*block_fft)/np.sum(block_fft)

    #take the mean of all of the audio file's block centroids and round to two decimal places
    mean_centroid_for_file = round(np.mean(centroid_list), 2)

    #print the mean spectral centroid for current file
    print(f'{mean_centroid_for_file} Hz')

    #append the mean spectral centroid to the genre mean spectral centroid dictionary
    mean_spectral_centroids[k].append(mean_centroid_for_file)
#print the dictionary of spectral centroids
print(mean_spectral_centroids)
```

    
    -----classical file 1-----
    ./music_dataFolder/classical/classical1.wav
    1440.18 Hz
    
    -----classical file 2-----
    ./music_dataFolder/classical/classical2.wav
    1446.49 Hz
    
    -----classical file 3-----
    ./music_dataFolder/classical/classical.wav
    1607.0 Hz
    
    -----classical file 4-----
    ./music_dataFolder/classical/copland.wav
    1872.75 Hz
    
    -----classical file 5-----
    ./music_dataFolder/classical/copland2.wav
    1471.75 Hz
    
    -----classical file 6-----
    ./music_dataFolder/classical/vlobos.wav
    1450.13 Hz
    
    -----classical file 7-----
    ./music_dataFolder/classical/brahms.wav
    1226.17 Hz
    
    -----classical file 8-----
    ./music_dataFolder/classical/debussy.wav
    1528.46 Hz
    
    -----classical file 9-----
    ./music_dataFolder/classical/bartok.wav
    1826.7 Hz
    
    -----jazz file 1-----
    ./music_dataFolder/jazz/ipanema.wav
    2404.56 Hz
    
    -----jazz file 2-----
    ./music_dataFolder/jazz/duke.wav
    1473.18 Hz
    
    -----jazz file 3-----
    ./music_dataFolder/jazz/moanin.wav
    1584.5 Hz
    
    -----jazz file 4-----
    ./music_dataFolder/jazz/russo.wav
    1921.62 Hz
    
    -----jazz file 5-----
    ./music_dataFolder/jazz/jazz1.wav
    2474.58 Hz
    
    -----jazz file 6-----
    ./music_dataFolder/jazz/mingus1.wav
    1742.44 Hz
    
    -----jazz file 7-----
    ./music_dataFolder/jazz/tony.wav
    2324.38 Hz
    
    -----jazz file 8-----
    ./music_dataFolder/jazz/misirlou.wav
    1782.58 Hz
    
    -----jazz file 9-----
    ./music_dataFolder/jazz/corea1.wav
    1696.76 Hz
    
    -----jazz file 10-----
    ./music_dataFolder/jazz/beat.wav
    2254.55 Hz
    
    -----jazz file 11-----
    ./music_dataFolder/jazz/georose.wav
    2633.23 Hz
    
    -----jazz file 12-----
    ./music_dataFolder/jazz/caravan.wav
    1219.48 Hz
    
    -----jazz file 13-----
    ./music_dataFolder/jazz/bmarsalis.wav
    2360.85 Hz
    
    -----jazz file 14-----
    ./music_dataFolder/jazz/mingus.wav
    1782.91 Hz
    
    -----jazz file 15-----
    ./music_dataFolder/jazz/unpoco.wav
    2434.07 Hz
    
    -----jazz file 16-----
    ./music_dataFolder/jazz/jazz.wav
    1859.11 Hz
    
    -----jazz file 17-----
    ./music_dataFolder/jazz/corea.wav
    1214.99 Hz
    
    -----rockblues file 1-----
    ./music_dataFolder/rockblues/rock2.wav
    2567.87 Hz
    
    -----rockblues file 2-----
    ./music_dataFolder/rockblues/hendrix.wav
    1874.48 Hz
    
    -----rockblues file 3-----
    ./music_dataFolder/rockblues/beatles.wav
    2091.79 Hz
    
    -----rockblues file 4-----
    ./music_dataFolder/rockblues/rock.wav
    2690.58 Hz
    
    -----rockblues file 5-----
    ./music_dataFolder/rockblues/blues.wav
    2154.8 Hz
    
    -----rockblues file 6-----
    ./music_dataFolder/rockblues/chaka.wav
    2479.06 Hz
    
    -----rockblues file 7-----
    ./music_dataFolder/rockblues/redhot.wav
    2673.32 Hz
    
    -----rockblues file 8-----
    ./music_dataFolder/rockblues/u2.wav
    2277.85 Hz
    
    -----rockblues file 9-----
    ./music_dataFolder/rockblues/led.wav
    2647.33 Hz
    
    -----rockblues file 10-----
    ./music_dataFolder/rockblues/eguitar.wav
    2570.5 Hz
    
    -----rockblues file 11-----
    ./music_dataFolder/rockblues/cure.wav
    2125.07 Hz
    
    -----speech file 1-----
    ./music_dataFolder/speech/teachers1.wav
    1560.3 Hz
    
    -----speech file 2-----
    ./music_dataFolder/speech/dialogue1.wav
    2211.36 Hz
    
    -----speech file 3-----
    ./music_dataFolder/speech/serbian.wav
    2714.72 Hz
    
    -----speech file 4-----
    ./music_dataFolder/speech/male.wav
    1884.36 Hz
    
    -----speech file 5-----
    ./music_dataFolder/speech/vegetables1.wav
    1728.24 Hz
    
    -----speech file 6-----
    ./music_dataFolder/speech/charles.wav
    2009.47 Hz
    
    -----speech file 7-----
    ./music_dataFolder/speech/voice.wav
    1902.53 Hz
    
    -----speech file 8-----
    ./music_dataFolder/speech/india.wav
    2257.41 Hz
    
    -----speech file 9-----
    ./music_dataFolder/speech/psychic.wav
    1543.64 Hz
    
    -----speech file 10-----
    ./music_dataFolder/speech/smoke1.wav
    2031.3 Hz
    
    -----speech file 11-----
    ./music_dataFolder/speech/vegetables.wav
    1656.39 Hz
    
    -----speech file 12-----
    ./music_dataFolder/speech/news1.wav
    2433.65 Hz
    
    -----speech file 13-----
    ./music_dataFolder/speech/fem_rock.wav
    2055.81 Hz
    
    -----speech file 14-----
    ./music_dataFolder/speech/allison.wav
    2536.18 Hz
    {'classical': [1440.18, 1446.49, 1607.0, 1872.75, 1471.75, 1450.13, 1226.17, 1528.46, 1826.7], 'jazz': [2404.56, 1473.18, 1584.5, 1921.62, 2474.58, 1742.44, 2324.38, 1782.58, 1696.76, 2254.55, 2633.23, 1219.48, 2360.85, 1782.91, 2434.07, 1859.11, 1214.99], 'rockblues': [2567.87, 1874.48, 2091.79, 2690.58, 2154.8, 2479.06, 2673.32, 2277.85, 2647.33, 2570.5, 2125.07], 'speech': [1560.3, 2211.36, 2714.72, 1884.36, 1728.24, 2009.47, 1902.53, 2257.41, 1543.64, 2031.3, 1656.39, 2433.65, 2055.81, 2536.18]}


## Feature 2: Standard deviation of RMS amplitude

The standard deviation of the RMS amplitude of each block is determined in order to analyze the variation of the frequency components around the average frequency magnitude. It is calculated by the following:

$v_{RMS,block}(i)=20log_{10}(\sqrt{\frac{\sum_{k=0}^{block\_size-1} |X(i, k)|^2}{block\_size}})$

<br>

$v_{RMS,SD}=\sqrt{\frac{\sum_{i=0}^{n\_blocks-1} (v_{RMS,block}(i) - v_{RMS,mean})^2}{n\_blocks}}$


```python
#define dictionary for calculated standard deviation of RMS values for each genre
std_rms_amp = {
    'classical': [],
    'jazz': [],
    'rockblues': [],
    'speech': []
}

#loop through dictionary of audio recording genres
for k, v in files.items():
  
  #loop through each recording for the corresponding genre
  for i in range(len(v)):

    #print basic information of audio file
    print()
    print("-----" + k + " file " + str(i+1) + "-----")
    print(v[i])
    
    #load audio recording through essentia
    song = es.MonoLoader(filename=v[i], sampleRate=Fs)()

    #get number of samples for the current audio recording
    n_samples = song.shape[0]

    #define basic parameters for block based analysis of recording
    block_size = 100000
    overlap = 0.25 #25% overlap between blocks
    block_step = (1-overlap)*block_size
    n_blocks = np.ceil(n_samples/block_step)

    #initialize zero np array to store RMS amplitude for each block
    rms_amp = np.zeros(int(n_blocks))

    #loop through all predefined blocks of audio recording
    for i in range(int(n_blocks)):

      #calculate start and end sample step for current block
      block_s = int(i*block_step)
      block_e = int(min(block_s + block_size, n_samples))

      #get audio data for the specific block
      block = song[block_s:block_e]

      #define number of samples for the current block
      Nsamples = block.shape[0]

      #calculate RMS amplitude for the current block
      rms = np.sqrt(np.mean(np.square(song[block_s:block_e])))

      #convert RMS amplitude units to decibels
      if rms<0.00001:
        rms = 0.00001
      db_rms = 20*np.log10(rms)

      #add calculated RMS amplitude to list of all block RMS amplitudes
      rms_amp[i] = db_rms
    
    #take the standard diviation of all blocks in the current audio file and round to two decimal places
    std_rms_for_file = round(np.std(rms_amp), 2)

    #print the standard deviation for the RMS ampltude of the audio file among all blocks
    print(f'{std_rms_for_file} dB')

    #append the standard deviation for the RMS ampltude of the audio file to the genre standard diviation RMS amplitude dictionary
    std_rms_amp[k].append(std_rms_for_file)
  
#print the dictionary of standard deviation RMS amplitude values among all files in all genres
print(std_rms_amp)
```

    
    -----classical file 1-----
    ./music_dataFolder/classical/classical1.wav
    2.81 dB
    
    -----classical file 2-----
    ./music_dataFolder/classical/classical2.wav
    2.19 dB
    
    -----classical file 3-----
    ./music_dataFolder/classical/classical.wav
    4.2 dB
    
    -----classical file 4-----
    ./music_dataFolder/classical/copland.wav
    1.57 dB
    
    -----classical file 5-----
    ./music_dataFolder/classical/copland2.wav
    3.86 dB
    
    -----classical file 6-----
    ./music_dataFolder/classical/vlobos.wav
    2.24 dB
    
    -----classical file 7-----
    ./music_dataFolder/classical/brahms.wav
    3.22 dB
    
    -----classical file 8-----
    ./music_dataFolder/classical/debussy.wav
    5.13 dB
    
    -----classical file 9-----
    ./music_dataFolder/classical/bartok.wav
    2.21 dB
    
    -----jazz file 1-----
    ./music_dataFolder/jazz/ipanema.wav
    2.25 dB
    
    -----jazz file 2-----
    ./music_dataFolder/jazz/duke.wav
    1.34 dB
    
    -----jazz file 3-----
    ./music_dataFolder/jazz/moanin.wav
    2.9 dB
    
    -----jazz file 4-----
    ./music_dataFolder/jazz/russo.wav
    1.76 dB
    
    -----jazz file 5-----
    ./music_dataFolder/jazz/jazz1.wav
    0.63 dB
    
    -----jazz file 6-----
    ./music_dataFolder/jazz/mingus1.wav
    3.77 dB
    
    -----jazz file 7-----
    ./music_dataFolder/jazz/tony.wav
    1.26 dB
    
    -----jazz file 8-----
    ./music_dataFolder/jazz/misirlou.wav
    0.81 dB
    
    -----jazz file 9-----
    ./music_dataFolder/jazz/corea1.wav
    0.78 dB
    
    -----jazz file 10-----
    ./music_dataFolder/jazz/beat.wav
    0.69 dB
    
    -----jazz file 11-----
    ./music_dataFolder/jazz/georose.wav
    0.28 dB
    
    -----jazz file 12-----
    ./music_dataFolder/jazz/caravan.wav
    0.9 dB
    
    -----jazz file 13-----
    ./music_dataFolder/jazz/bmarsalis.wav
    0.29 dB
    
    -----jazz file 14-----
    ./music_dataFolder/jazz/mingus.wav
    2.78 dB
    
    -----jazz file 15-----
    ./music_dataFolder/jazz/unpoco.wav
    1.24 dB
    
    -----jazz file 16-----
    ./music_dataFolder/jazz/jazz.wav
    1.14 dB
    
    -----jazz file 17-----
    ./music_dataFolder/jazz/corea.wav
    2.26 dB
    
    -----rockblues file 1-----
    ./music_dataFolder/rockblues/rock2.wav
    0.29 dB
    
    -----rockblues file 2-----
    ./music_dataFolder/rockblues/hendrix.wav
    0.5 dB
    
    -----rockblues file 3-----
    ./music_dataFolder/rockblues/beatles.wav
    1.41 dB
    
    -----rockblues file 4-----
    ./music_dataFolder/rockblues/rock.wav
    0.39 dB
    
    -----rockblues file 5-----
    ./music_dataFolder/rockblues/blues.wav
    2.28 dB
    
    -----rockblues file 6-----
    ./music_dataFolder/rockblues/chaka.wav
    0.65 dB
    
    -----rockblues file 7-----
    ./music_dataFolder/rockblues/redhot.wav
    0.28 dB
    
    -----rockblues file 8-----
    ./music_dataFolder/rockblues/u2.wav
    0.91 dB
    
    -----rockblues file 9-----
    ./music_dataFolder/rockblues/led.wav
    0.79 dB
    
    -----rockblues file 10-----
    ./music_dataFolder/rockblues/eguitar.wav
    0.51 dB
    
    -----rockblues file 11-----
    ./music_dataFolder/rockblues/cure.wav
    0.42 dB
    
    -----speech file 1-----
    ./music_dataFolder/speech/teachers1.wav
    0.52 dB
    
    -----speech file 2-----
    ./music_dataFolder/speech/dialogue1.wav
    0.68 dB
    
    -----speech file 3-----
    ./music_dataFolder/speech/serbian.wav
    3.32 dB
    
    -----speech file 4-----
    ./music_dataFolder/speech/male.wav
    1.12 dB
    
    -----speech file 5-----
    ./music_dataFolder/speech/vegetables1.wav
    0.8 dB
    
    -----speech file 6-----
    ./music_dataFolder/speech/charles.wav
    4.32 dB
    
    -----speech file 7-----
    ./music_dataFolder/speech/voice.wav
    0.82 dB
    
    -----speech file 8-----
    ./music_dataFolder/speech/india.wav
    2.25 dB
    
    -----speech file 9-----
    ./music_dataFolder/speech/psychic.wav
    0.59 dB
    
    -----speech file 10-----
    ./music_dataFolder/speech/smoke1.wav
    1.29 dB
    
    -----speech file 11-----
    ./music_dataFolder/speech/vegetables.wav
    0.89 dB
    
    -----speech file 12-----
    ./music_dataFolder/speech/news1.wav
    0.32 dB
    
    -----speech file 13-----
    ./music_dataFolder/speech/fem_rock.wav
    2.65 dB
    
    -----speech file 14-----
    ./music_dataFolder/speech/allison.wav
    1.17 dB
    {'classical': [2.81, 2.19, 4.2, 1.57, 3.86, 2.24, 3.22, 5.13, 2.21], 'jazz': [2.25, 1.34, 2.9, 1.76, 0.63, 3.77, 1.26, 0.81, 0.78, 0.69, 0.28, 0.9, 0.29, 2.78, 1.24, 1.14, 2.26], 'rockblues': [0.29, 0.5, 1.41, 0.39, 2.28, 0.65, 0.28, 0.91, 0.79, 0.51, 0.42], 'speech': [0.52, 0.68, 3.32, 1.12, 0.8, 4.32, 0.82, 2.25, 0.59, 1.29, 0.89, 0.32, 2.65, 1.17]}


## Feature 3: Mean Spectral Rolloff

The mean spectral rolloff is found to determine the bandwidth in which a certain percentage of the frequency spectrum resides. A value of 90% is chosen, and the calculation is implemented as follows:

$v_{SR,block}(i)=m|_{\sum_{k=0}^m|X(i,k)|\text{ = }P\cdot\sum_{k=0}^{block\_size-1}|X(i,k)|}$

<br>

$v_{SR,mean}=\frac{\sum_{i=0}^{n\_blocks-1} v_{SR,block}(i)}{n\_blocks}$


```python
#define dictionary for calculated spectral rolloff values for each genre
spectral_rolloff = {
    'classical': [],
    'jazz': [],
    'rockblues': [],
    'speech': []
}

#define the bandwidth percentage when calculating the spectral rolloff
BANDWIDTH_PERCENTAGE = 0.90

#loop through dictionary of audio recording genres
for k, v in files.items():
  
  #loop through each recording for the corresponding genre
  for i in range(len(v)):

    #print basic information of audio file
    print()
    print("-----" + k + " file " + str(i+1) + "-----")
    print(v[i])
    
    #load audio recording through essentia
    song = es.MonoLoader(filename=v[i], sampleRate=Fs)()

    #get number of samples for the current audio recording
    n_samples = song.shape[0]

    #define basic parameters for block based analysis of recording
    block_size = 100000
    overlap = 0.25 #25% overlap between blocks
    block_step = (1-overlap)*block_size
    n_blocks = np.ceil(n_samples/block_step)

    #initialize zero np array to store spectral rolloff for each block
    rolloff_list = np.zeros(int(n_blocks))
    rolloff = []

    #loop through all predefined blocks of audio recording
    for i in range(int(n_blocks)):

      #calculate start and end sample step for current block
      block_s = int(i*block_step)
      block_e = int(min(block_s + block_size, n_samples))

      #calculate audio data for the specific block
      block = song[block_s:block_e]

      #define number of samples for the current block
      Nsamples = block.shape[0]

      ff = np.arange(0,int(Fs),Fs/Nsamples)

      #calculate the fast fourier transform for the current block
      block_fft = np.abs(np.fft.fft(block))

      #only display half of the samples of the audio file for more accurate FFT
      ff = ff[:int(Nsamples/2)]
      block_fft = block_fft[:int(Nsamples/2)]

      #sum over the entire block's spectral range of FFT
      total_sum = np.sum(block_fft)


      #variable to hold the accumulated magnitude of the rolloff calculation
      accumilated_mag = 0
      
      #calculate the spectral rolloff of the block
      for increment, value in enumerate(block_fft):
        accumilated_mag += value
        if accumilated_mag >= BANDWIDTH_PERCENTAGE*total_sum:
          rolloff.append(increment)
          break

    #calculate the scaled percentage for the rolloff calculation
    scaled_percentage = np.average(rolloff) / len(block_fft)

    #calculate the rolloff frequency for the audio file, serving as the mean spectral rolloff
    rolloff_freq = round(scaled_percentage * Fs, 2)

    #print the mean spectral rolloff for the audio file
    print(f'{rolloff_freq} Hz')

    #append the mean spectral rolloff to the dictionary holding the genre spectral rolloffs 
    spectral_rolloff[k].append(rolloff_freq)

#print the dictionary of spectral rolloffs for all genres
print(spectral_rolloff)
```

    
    -----classical file 1-----
    ./music_dataFolder/classical/classical1.wav
    13766.86 Hz
    
    -----classical file 2-----
    ./music_dataFolder/classical/classical2.wav
    12165.68 Hz
    
    -----classical file 3-----
    ./music_dataFolder/classical/classical.wav
    17048.32 Hz
    
    -----classical file 4-----
    ./music_dataFolder/classical/copland.wav
    14536.87 Hz
    
    -----classical file 5-----
    ./music_dataFolder/classical/copland2.wav
    13685.7 Hz
    
    -----classical file 6-----
    ./music_dataFolder/classical/vlobos.wav
    12723.16 Hz
    
    -----classical file 7-----
    ./music_dataFolder/classical/brahms.wav
    10509.17 Hz
    
    -----classical file 8-----
    ./music_dataFolder/classical/debussy.wav
    13717.35 Hz
    
    -----classical file 9-----
    ./music_dataFolder/classical/bartok.wav
    16452.46 Hz
    
    -----jazz file 1-----
    ./music_dataFolder/jazz/ipanema.wav
    22320.42 Hz
    
    -----jazz file 2-----
    ./music_dataFolder/jazz/duke.wav
    13588.82 Hz
    
    -----jazz file 3-----
    ./music_dataFolder/jazz/moanin.wav
    14277.89 Hz
    
    -----jazz file 4-----
    ./music_dataFolder/jazz/russo.wav
    21317.25 Hz
    
    -----jazz file 5-----
    ./music_dataFolder/jazz/jazz1.wav
    26464.19 Hz
    
    -----jazz file 6-----
    ./music_dataFolder/jazz/mingus1.wav
    18481.78 Hz
    
    -----jazz file 7-----
    ./music_dataFolder/jazz/tony.wav
    25618.32 Hz
    
    -----jazz file 8-----
    ./music_dataFolder/jazz/misirlou.wav
    19922.18 Hz
    
    -----jazz file 9-----
    ./music_dataFolder/jazz/corea1.wav
    19931.67 Hz
    
    -----jazz file 10-----
    ./music_dataFolder/jazz/beat.wav
    23953.65 Hz
    
    -----jazz file 11-----
    ./music_dataFolder/jazz/georose.wav
    32395.23 Hz
    
    -----jazz file 12-----
    ./music_dataFolder/jazz/caravan.wav
    13316.36 Hz
    
    -----jazz file 13-----
    ./music_dataFolder/jazz/bmarsalis.wav
    27292.29 Hz
    
    -----jazz file 14-----
    ./music_dataFolder/jazz/mingus.wav
    19059.16 Hz
    
    -----jazz file 15-----
    ./music_dataFolder/jazz/unpoco.wav
    26639.77 Hz
    
    -----jazz file 16-----
    ./music_dataFolder/jazz/jazz.wav
    17194.3 Hz
    
    -----jazz file 17-----
    ./music_dataFolder/jazz/corea.wav
    11358.3 Hz
    
    -----rockblues file 1-----
    ./music_dataFolder/rockblues/rock2.wav
    27690.92 Hz
    
    -----rockblues file 2-----
    ./music_dataFolder/rockblues/hendrix.wav
    17733.0 Hz
    
    -----rockblues file 3-----
    ./music_dataFolder/rockblues/beatles.wav
    22248.25 Hz
    
    -----rockblues file 4-----
    ./music_dataFolder/rockblues/rock.wav
    26637.01 Hz
    
    -----rockblues file 5-----
    ./music_dataFolder/rockblues/blues.wav
    23150.66 Hz
    
    -----rockblues file 6-----
    ./music_dataFolder/rockblues/chaka.wav
    26092.91 Hz
    
    -----rockblues file 7-----
    ./music_dataFolder/rockblues/redhot.wav
    27187.65 Hz
    
    -----rockblues file 8-----
    ./music_dataFolder/rockblues/u2.wav
    24764.7 Hz
    
    -----rockblues file 9-----
    ./music_dataFolder/rockblues/led.wav
    16443.69 Hz
    
    -----rockblues file 10-----
    ./music_dataFolder/rockblues/eguitar.wav
    27060.66 Hz
    
    -----rockblues file 11-----
    ./music_dataFolder/rockblues/cure.wav
    22527.34 Hz
    
    -----speech file 1-----
    ./music_dataFolder/speech/teachers1.wav
    12913.24 Hz
    
    -----speech file 2-----
    ./music_dataFolder/speech/dialogue1.wav
    21346.34 Hz
    
    -----speech file 3-----
    ./music_dataFolder/speech/serbian.wav
    25442.94 Hz
    
    -----speech file 4-----
    ./music_dataFolder/speech/male.wav
    20592.35 Hz
    
    -----speech file 5-----
    ./music_dataFolder/speech/vegetables1.wav
    14059.32 Hz
    
    -----speech file 6-----
    ./music_dataFolder/speech/charles.wav
    20900.95 Hz
    
    -----speech file 7-----
    ./music_dataFolder/speech/voice.wav
    17467.48 Hz
    
    -----speech file 8-----
    ./music_dataFolder/speech/india.wav
    24696.1 Hz
    
    -----speech file 9-----
    ./music_dataFolder/speech/psychic.wav
    12213.25 Hz
    
    -----speech file 10-----
    ./music_dataFolder/speech/smoke1.wav
    19417.99 Hz
    
    -----speech file 11-----
    ./music_dataFolder/speech/vegetables.wav
    13798.4 Hz
    
    -----speech file 12-----
    ./music_dataFolder/speech/news1.wav
    25928.66 Hz
    
    -----speech file 13-----
    ./music_dataFolder/speech/fem_rock.wav
    20577.55 Hz
    
    -----speech file 14-----
    ./music_dataFolder/speech/allison.wav
    25556.77 Hz
    {'classical': [13766.86, 12165.68, 17048.32, 14536.87, 13685.7, 12723.16, 10509.17, 13717.35, 16452.46], 'jazz': [22320.42, 13588.82, 14277.89, 21317.25, 26464.19, 18481.78, 25618.32, 19922.18, 19931.67, 23953.65, 32395.23, 13316.36, 27292.29, 19059.16, 26639.77, 17194.3, 11358.3], 'rockblues': [27690.92, 17733.0, 22248.25, 26637.01, 23150.66, 26092.91, 27187.65, 24764.7, 16443.69, 27060.66, 22527.34], 'speech': [12913.24, 21346.34, 25442.94, 20592.35, 14059.32, 20900.95, 17467.48, 24696.1, 12213.25, 19417.99, 13798.4, 25928.66, 20577.55, 25556.77]}


## Feature 4: Mean Spectral Flatness

The specral flatness is defined as the ratio of the signal noisiness to signal tonalness. Higher values of spectral flatness would attribute to a noisier spectrum, while lower values will suggest a more tonal spectrum. The logarithmic magnitude spectrum is used:

$v_{SF,block}(i)=\frac{exp(\sum_{i=0}^{block\_size-1} log(|X(i, k)|))}{\sum_{i=0}^{block\_size-1} |X(i, k)|}$

<br>

$v_{SF,mean}=\frac{\sum_{i=0}^{n\_blocks-1} v_{SF,block}(i)}{n\_blocks}$


```python
#define dictionary for calculated spectral flatness values for each genre
spectral_flatness = {
    'classical': [],
    'jazz': [],
    'rockblues': [],
    'speech': []
}

#loop through dictionary of audio recording genres
for k, v in files.items():
  
  #loop through each recording for the corresponding genre
  for i in range(len(v)):

    #print basic information of audio file
    print()
    print("-----" + k + " file " + str(i+1) + "-----")
    print(v[i])
    
    #load audio recording through essentia
    song = es.MonoLoader(filename=v[i], sampleRate=Fs)()

    #get number of samples for the current audio recording
    n_samples = song.shape[0]

    #define basic parameters for block based analysis of recording
    block_size = 100000
    overlap = 0.25
    block_step = (1-overlap)*block_size
    n_blocks = np.ceil(n_samples/block_step)

    #initialize zero np array to store spectral flatness for each block
    flatness_list = np.zeros(int(n_blocks))
    flatness = []

    #loop through all predefined blocks of audio recording
    for i in range(int(n_blocks)):

      #calculate start and end sample step for current block
      block_s = int(i*block_step)
      block_e = int(min(block_s + block_size, n_samples))

      #calculate audio data for the specific block
      block = song[block_s:block_e]

      #define number of samples for the current block
      Nsamples = block.shape[0]

      ff = np.arange(0,int(Fs),Fs/Nsamples)

      #calculate the fast fourier transform for the current block
      block_fft = np.abs(np.fft.fft(block))

      #only display half of the samples of the audio file to avoid repeating FFT
      ff = ff[:int(Nsamples/2)]
      block_fft = block_fft[:int(Nsamples/2)]

      #calculate log magnitude for the block's FFT
      log_mag = np.log(block_fft)

      #calculate the sum along the entire log magnitude of the FFT
      log_sum = np.sum(log_mag)

      #define exponent section of spectral flatness calculation
      exp = np.exp(1/len(block_fft)*log_sum)

      #calculate the sum along the entire frequency amplitudes of the FFT
      total_sum = np.sum(block_fft)

      #calculate current block's spectral flatness and append to flatness list
      flatness.append(exp / (1/len(block_fft)*total_sum))

    #calculate the mean spectral flatness along all blocks of the audio recording
    mean_flatness = round(np.average(flatness), 6)

    #print the current recording's mean spectral flatness
    print(mean_flatness)

    #append the mean spectral flatness to the genre mean spectral flatness dictionary
    spectral_flatness[k].append(mean_flatness)

#print the dictionary of mean spectral flatness
print(spectral_flatness)
```

    
    -----classical file 1-----
    ./music_dataFolder/classical/classical1.wav
    0.050081
    
    -----classical file 2-----
    ./music_dataFolder/classical/classical2.wav
    0.067264
    
    -----classical file 3-----
    ./music_dataFolder/classical/classical.wav
    0.070513
    
    -----classical file 4-----
    ./music_dataFolder/classical/copland.wav
    0.053465
    
    -----classical file 5-----
    ./music_dataFolder/classical/copland2.wav
    0.050819
    
    -----classical file 6-----
    ./music_dataFolder/classical/vlobos.wav
    0.043345
    
    -----classical file 7-----
    ./music_dataFolder/classical/brahms.wav
    0.041181
    
    -----classical file 8-----
    ./music_dataFolder/classical/debussy.wav
    0.050647
    
    -----classical file 9-----
    ./music_dataFolder/classical/bartok.wav
    0.046758
    
    -----jazz file 1-----
    ./music_dataFolder/jazz/ipanema.wav
    0.061072
    
    -----jazz file 2-----
    ./music_dataFolder/jazz/duke.wav
    0.044529
    
    -----jazz file 3-----
    ./music_dataFolder/jazz/moanin.wav
    0.05593
    
    -----jazz file 4-----
    ./music_dataFolder/jazz/russo.wav
    0.070319
    
    -----jazz file 5-----
    ./music_dataFolder/jazz/jazz1.wav
    0.066865
    
    -----jazz file 6-----
    ./music_dataFolder/jazz/mingus1.wav
    0.060528
    
    -----jazz file 7-----
    ./music_dataFolder/jazz/tony.wav
    0.07055
    
    -----jazz file 8-----
    ./music_dataFolder/jazz/misirlou.wav
    0.055627
    
    -----jazz file 9-----
    ./music_dataFolder/jazz/corea1.wav
    0.065826
    
    -----jazz file 10-----
    ./music_dataFolder/jazz/beat.wav
    0.066172
    
    -----jazz file 11-----
    ./music_dataFolder/jazz/georose.wav
    0.075322
    
    -----jazz file 12-----
    ./music_dataFolder/jazz/caravan.wav
    0.048848
    
    -----jazz file 13-----
    ./music_dataFolder/jazz/bmarsalis.wav
    0.06751
    
    -----jazz file 14-----
    ./music_dataFolder/jazz/mingus.wav
    0.06727
    
    -----jazz file 15-----
    ./music_dataFolder/jazz/unpoco.wav
    0.069288
    
    -----jazz file 16-----
    ./music_dataFolder/jazz/jazz.wav
    0.042168
    
    -----jazz file 17-----
    ./music_dataFolder/jazz/corea.wav
    0.072225
    
    -----rockblues file 1-----
    ./music_dataFolder/rockblues/rock2.wav
    0.078339
    
    -----rockblues file 2-----
    ./music_dataFolder/rockblues/hendrix.wav
    0.044682
    
    -----rockblues file 3-----
    ./music_dataFolder/rockblues/beatles.wav
    0.101521
    
    -----rockblues file 4-----
    ./music_dataFolder/rockblues/rock.wav
    0.072254
    
    -----rockblues file 5-----
    ./music_dataFolder/rockblues/blues.wav
    0.057363
    
    -----rockblues file 6-----
    ./music_dataFolder/rockblues/chaka.wav
    0.065912
    
    -----rockblues file 7-----
    ./music_dataFolder/rockblues/redhot.wav
    0.060787
    
    -----rockblues file 8-----
    ./music_dataFolder/rockblues/u2.wav
    0.076254
    
    -----rockblues file 9-----
    ./music_dataFolder/rockblues/led.wav
    0.049484
    
    -----rockblues file 10-----
    ./music_dataFolder/rockblues/eguitar.wav
    0.063069
    
    -----rockblues file 11-----
    ./music_dataFolder/rockblues/cure.wav
    0.059316
    
    -----speech file 1-----
    ./music_dataFolder/speech/teachers1.wav
    0.037083
    
    -----speech file 2-----
    ./music_dataFolder/speech/dialogue1.wav
    0.050817
    
    -----speech file 3-----
    ./music_dataFolder/speech/serbian.wav
    0.07337
    
    -----speech file 4-----
    ./music_dataFolder/speech/male.wav
    0.04728
    
    -----speech file 5-----
    ./music_dataFolder/speech/vegetables1.wav
    0.044283
    
    -----speech file 6-----
    ./music_dataFolder/speech/charles.wav
    0.047723
    
    -----speech file 7-----
    ./music_dataFolder/speech/voice.wav
    0.048353
    
    -----speech file 8-----
    ./music_dataFolder/speech/india.wav
    0.062586
    
    -----speech file 9-----
    ./music_dataFolder/speech/psychic.wav
    0.032671
    
    -----speech file 10-----
    ./music_dataFolder/speech/smoke1.wav
    0.051657
    
    -----speech file 11-----
    ./music_dataFolder/speech/vegetables.wav
    0.04278
    
    -----speech file 12-----
    ./music_dataFolder/speech/news1.wav
    0.071796
    
    -----speech file 13-----
    ./music_dataFolder/speech/fem_rock.wav
    0.053993
    
    -----speech file 14-----
    ./music_dataFolder/speech/allison.wav
    0.063278
    {'classical': [0.050081, 0.067264, 0.070513, 0.053465, 0.050819, 0.043345, 0.041181, 0.050647, 0.046758], 'jazz': [0.061072, 0.044529, 0.05593, 0.070319, 0.066865, 0.060528, 0.07055, 0.055627, 0.065826, 0.066172, 0.075322, 0.048848, 0.06751, 0.06727, 0.069288, 0.042168, 0.072225], 'rockblues': [0.078339, 0.044682, 0.101521, 0.072254, 0.057363, 0.065912, 0.060787, 0.076254, 0.049484, 0.063069, 0.059316], 'speech': [0.037083, 0.050817, 0.07337, 0.04728, 0.044283, 0.047723, 0.048353, 0.062586, 0.032671, 0.051657, 0.04278, 0.071796, 0.053993, 0.063278]}


# **Feature Plots**

Plotting Mean Spectral Centroid vs. Mean Spectral Rolloff


```python
plt.figure()
plt.title('Mean Spectral Centroid vs. Mean Spectral Rolloff')
plt.scatter(spectral_rolloff['classical'], mean_spectral_centroids['classical'], label='classical')
plt.scatter(spectral_rolloff['jazz'], mean_spectral_centroids['jazz'], label='jazz')
plt.scatter(spectral_rolloff['rockblues'], mean_spectral_centroids['rockblues'], label='rockblues')
plt.scatter(spectral_rolloff['speech'], mean_spectral_centroids['speech'], label='speech')
plt.xlabel('Mean Spectral Rolloff (Hz)')
plt.ylabel('Mean Spectral Centroid (Hz)')
plt.legend()
plt.grid()
plt.show()
```


    
![png](signal_feature_extraction_project_files/signal_feature_extraction_project_21_0.png)
    


Plotting Mean Spectral Centroid vs. Mean Spectral Flatness


```python
plt.figure()
plt.title('Mean Spectral Centroid vs. Mean Spectral Flatness')
plt.scatter(spectral_flatness['classical'], mean_spectral_centroids['classical'], label='classical')
plt.scatter(spectral_flatness['jazz'], mean_spectral_centroids['jazz'], label='jazz')
plt.scatter(spectral_flatness['rockblues'], mean_spectral_centroids['rockblues'], label='rockblues')
plt.scatter(spectral_flatness['speech'], mean_spectral_centroids['speech'], label='speech')
plt.xlabel('Mean Spectral Flatness')
plt.ylabel('Mean Spectral Centroid (Hz)')
plt.legend()
plt.grid()
plt.show()
```


    
![png](signal_feature_extraction_project_files/signal_feature_extraction_project_23_0.png)
    


Plotting Mean Spectral Centroid vs. Standard Deviation of RMS Amplitude


```python
plt.figure()
plt.title('Mean Spectral Centroid vs. Standard Deviation of RMS Amplitude')
plt.scatter(std_rms_amp['classical'], mean_spectral_centroids['classical'], label='classical')
plt.scatter(std_rms_amp['jazz'], mean_spectral_centroids['jazz'], label='jazz')
plt.scatter(std_rms_amp['rockblues'], mean_spectral_centroids['rockblues'], label='rockblues')
plt.scatter(std_rms_amp['speech'], mean_spectral_centroids['speech'], label='speech')
plt.xlabel('Standard Deviation of RMS Amplitude (Hz)')
plt.ylabel('Mean Spectral Centroid (Hz)')
plt.legend()
plt.grid()
plt.show()
```


    
![png](signal_feature_extraction_project_files/signal_feature_extraction_project_25_0.png)
    


Plotting Mean Spectral Rolloff vs. Mean Spectral Flatness


```python
plt.figure()
plt.title('Mean Spectral Rolloff vs. Mean Spectral Flatness')
plt.scatter(spectral_rolloff['classical'], spectral_flatness['classical'], label='classical')
plt.scatter(spectral_rolloff['jazz'], spectral_flatness['jazz'], label='jazz')
plt.scatter(spectral_rolloff['rockblues'], spectral_flatness['rockblues'], label='rockblues')
plt.scatter(spectral_rolloff['speech'], spectral_flatness['speech'], label='speech')
plt.xlabel('Mean Spectral Rolloff (Hz)')
plt.ylabel('Mean Spectral Flatness')
plt.legend()
plt.grid()
plt.show()
```


    
![png](signal_feature_extraction_project_files/signal_feature_extraction_project_27_0.png)
    


Plotting Mean Spectral Rolloff vs. Standard Deviation of RMS Amplitude


```python
plt.figure()
plt.title('Mean Spectral Rolloff vs. Standard Deviation of RMS Amplitude')
plt.scatter(std_rms_amp['classical'], spectral_rolloff['classical'], label='classical')
plt.scatter(std_rms_amp['jazz'], spectral_rolloff['jazz'], label='jazz')
plt.scatter(std_rms_amp['rockblues'], spectral_rolloff['rockblues'], label='rockblues')
plt.scatter(std_rms_amp['speech'], spectral_rolloff['speech'], label='speech')
plt.ylabel('Mean Spectral Rolloff (Hz)')
plt.xlabel('Standard Deviation of RMS Amplitude (Hz)')
plt.legend()
plt.grid()
plt.show()
```


    
![png](signal_feature_extraction_project_files/signal_feature_extraction_project_29_0.png)
    


Plotting Mean Spectral Flatness vs. Standard Deviation of RMS Amplitude


```python
plt.figure()
plt.title('Mean Spectral Flatness vs. Standard Deviation of RMS Amplitude')
plt.scatter(std_rms_amp['classical'], spectral_flatness['classical'], label='classical')
plt.scatter(std_rms_amp['jazz'], spectral_flatness['jazz'], label='jazz')
plt.scatter(std_rms_amp['rockblues'], spectral_flatness['rockblues'], label='rockblues')
plt.scatter(std_rms_amp['speech'], spectral_flatness['speech'], label='speech')
plt.ylabel('Mean Spectral Flatness')
plt.xlabel('Standard Deviation of RMS Amplitude (Hz)')
plt.legend()
plt.grid()
plt.show()
```


    
![png](signal_feature_extraction_project_files/signal_feature_extraction_project_31_0.png)
    


# **Discussion**

The plotted features provide varying levels of separation based on genre. In summary, the rockblues genre provided a decent separation across all features, while jazz was never able to be adequately differentiated. Classical was able to cluster well in the spectral centroid and spectral rolloff features, and speech was only distinguishable in the spectral flatness feature.





<br>


The plotted spectral centroid vs. spectral rolloff plot demonstrates good clustering for the classical and rockblues genres, but also features scattered jazz and speech points. Figure 1 shows the centroid vs. spectral rolloff plot with classical and rockblues points.

<center>

![sc1.png](data:image/png;base64,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)

Figure 1. Centroid vs. Rolloff with classical and bluesrock points.</center>

<br>


Spectral flatness, which provides decent clustering for classical, rockblues, and speech, shows scattered data for jazz. When spectral flatness is plotted against spectral centroid or spectral rolloff, it is apparent that classical, rockblues, and speech each provide a somewhat distinguishable grouping. If for example jazz was removed, a KNN implementation may be used for future data classification. Figure 2 shows these two plots without the jazz data points.

<center>

![image.png](data:image/png;base64,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)

Figure 2. Example feature plots without the jazz data set.</center>
