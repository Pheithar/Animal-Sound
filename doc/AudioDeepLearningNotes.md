# Digital Representation of sound.
Sound is digitized by turning the signal into a series of numbers, measuring the amplitude of the sound at fixed intervals of time called **sample rate**. A common sample rate is 44100 samples per second. If the sound clip was 10 seconds long, thats 441.000 samples.

# Preparing audio data for deep learning
The common approach is to convert the audio data into images (spectograms) and using standard CNN architecture to process those images.

## Spectograms
Spectograms are another way of showing an audio signal. Where the waveform plot shows the amplitude of the sound at a given time, the spectogram shows the intensity of each frequency at each time slot. Often times, the frequencies are split into bins instead of unique frequencies.

## Data Augmentation

