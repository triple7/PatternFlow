# TTSGAN
A tf/keras implementation of the hi-fi TTS GAN text to speech generation using Generative Adversarial Networks (https://arxiv.org/pdf/1909.11646.pdf)

## Abstract

The original paper, released in 2019, determined an accuracy close to the best auto-regressors such as WaveNet (van den Oord et al. ,  2016), sampleRNN (Mehr i  et al. ,  2017) and waveRNN (Kalchbrenner et al. ,  2018).

Though resulting in high quality output, a notable issue with these neural networks is that audio samples are analysed sequentially, and do not have the ability to perform parallel computation of larger sample frames.
	Furthermore, these models use raw waveform data, modelling the amplitude of the waveform as data is processed.

The TTSGAN leverages the significant advances made by GAN models in the image domain, but places focus on using the same categorisation methods used for audio, by extracting conditional distribution features made possible by transforming time based data to its spectral domain. 

## MEL spectrum

The MEL spectrogram conversion used in this model is different from the classic spectrogram (https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0), in normalising distances across frequency modes.
In classic spectrograms, distances between frequencies grow larger as a ratio of the frequency itself.
The MEL spectrogram returns equidistant distance features across frequencies.

# Structure

The TTSGAN model consists of 2 components, similar to classic GANs:
1. Generator: a 7 G block (convolutions, and spectral batch normalisation layers)
2. Discriminator (5 temporal distance layers with a custom reflective padding of the original sample)
3. TTSGAN: alternates between generator and discriminator 

The model's general loss function uses multi-resolution spectral convergence to compute the distance between estimator and truth per bin in the MEL spectrogram.

## Optimiser 

The rollout decay optimiser is used, such that as 1 < x < 50, the learning rate remains static and > 50, the learning rate decreases.

## Dependencies

The install_dependencies.sh script will download necessary modules for audio sample conversion, youtube playlist downloading, and tensorflow addons, such as spectral normalisation.

## Downloading data samples

Use loadData.sh [playlist_URL] where Playlist_URL is a youtube playlist, which will be downloaded in ./data, and uses the youtube automatic transcription feature to associate sample frames with the transcription segment, under ./data/transcripts and ./data/processed.

Note: Some playlists do not have the transcripts. 

## Output

Check points output intermittent generated audio in ./generated

>