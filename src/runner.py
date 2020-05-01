from src.preprocessing.main import run as run_preprocessing
from src.training.TacoTrainer import run as run_tacotron_training
from src.training.GTASynthesizer import run as run_gta
from src.training.WavTraining import run as run_wav_training
from src.synthesis.TacoSynthesizer import run as run_tacotron_synth
from src.synthesis.WaveNetSynthesizer import run as run_wavnet_synth

def run():
  #run_preprocessing()
  print("##### Taco Training #######")
  #run_tacotron_training()
  print("##### GTA Synthesis #######")
  run_gta()
  print("##### WaveNet Training #######")
  run_wav_training()
  print("##### Taco Synthesis #######")
  run_tacotron_synth()
  print("##### WaveNet Synthesis #######")
  run_wavnet_synth()

if __name__ == "__main__":
  run()