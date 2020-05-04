Pre-trained models and audio samples will be added at a later date. You can however check some primary insights of the model performance (at early stages of training) [here](https://github.com/Rayhane-mamah/Tacotron-2/issues/4#issuecomment-378741465). THIS IS VERY OUTDATED, I WILL UPDATE THIS SOON

To have an in-depth exploration of the model architecture, training procedure and preprocessing logic, refer to [our wiki](https://github.com/Rayhane-mamah/Tacotron-2/wiki)

To pick optimal fft parameters, I have made a **griffin_lim_synthesis_tool** notebook that you can use to invert real extracted mel/linear spectrograms and choose how good your preprocessing is. All other options are well explained in the **hparams.py** and have meaningful names so that you can try multiple things with them.

For the spectrogram prediction network (separately), there are **three types** of mel spectrograms synthesis:

- **Evaluation** (synthesis on custom sentences). This is what we'll usually use after having a full end to end model.

> python synthesize.py --model='Tacotron'

- **Natural synthesis** (let the model make predictions alone by feeding last decoder output to the next time step).

> python synthesize.py --model='Tacotron' --mode='synthesis' --GTA=False


- **Ground Truth Aligned synthesis** (DEFAULT: the model is assisted by true labels in a teacher forcing manner). This synthesis method is used when predicting mel spectrograms used to train the wavenet vocoder. (yields better results as stated in the paper)

> python synthesize.py --model='Tacotron' --mode='synthesis' --GTA=True

Synthesizing the **waveforms** conditionned on previously synthesized Mel-spectrograms (separately) can be done with:

> python synthesize.py --model='WaveNet'
