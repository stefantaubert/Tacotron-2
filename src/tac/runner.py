from src.tac.preprocessing.main import run as run_preprocessing
from src.tac.training.tacotron_training import run as run_tacotron_training
from src.tac.training.tacotron_gta_synthesis import run as run_gta
from src.tac.training.wav_training import run as run_wav_training
from src.tac.synthesis.tacotron_eval import run as run_tacotron_synth
from src.tac.synthesis.wavenet_synthesis import run as run_wavnet_synth

def run(testrun: bool = False):
  run_preprocessing()
  print("##### Taco Training #######")
  run_tacotron_training(testrun)
  print("##### GTA Synthesis #######")
  run_gta()
  print("##### WaveNet Training #######")
  run_wav_training(testrun)
  print("##### Taco Synthesis #######")
  run_tacotron_synth()
  print("##### WaveNet Synthesis #######")
  run_wavnet_synth()

if __name__ == "__main__":
  run(testrun=True)