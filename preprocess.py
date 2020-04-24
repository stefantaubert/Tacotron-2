import argparse
import os
from multiprocessing import cpu_count

from datasets import preprocessor
from hparams import hparams
from tqdm import tqdm


def preprocess(args, input_folders, out_dir, hparams):
	mel_dir = os.path.join(out_dir, 'mels')
	wav_dir = os.path.join(out_dir, 'audio')
	linear_dir = os.path.join(out_dir, 'linear')
	os.makedirs(mel_dir, exist_ok=True)
	os.makedirs(wav_dir, exist_ok=True)
	os.makedirs(linear_dir, exist_ok=True)
	metadata = preprocessor.build_from_path(hparams, input_folders, mel_dir, linear_dir, wav_dir, args.n_jobs, tqdm=tqdm)
	write_metadata(metadata, out_dir)

def write_metadata(metadata, out_dir):
	with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
		for m in metadata:
			f.write('|'.join([str(x) for x in m]) + '\n')
	mel_frames = sum([int(m[4]) for m in metadata])
	timesteps = sum([int(m[3]) for m in metadata])
	sr = hparams.sample_rate
	hours = timesteps / sr / 3600
	print('Write {} utterances, {} mel frames, {} audio timesteps, ({:.2f} hours)'.format(
		len(metadata), mel_frames, timesteps, hours))
	print('Max input length (text chars): {}'.format(max(len(m[5]) for m in metadata)))
	print('Max mel frames length: {}'.format(max(int(m[4]) for m in metadata)))
	print('Max audio timesteps length: {}'.format(max(m[3] for m in metadata)))

def _get_mel_dir(caching_dir: str) => str:
	''' the directory to write the mel spectograms into '''
	path = os.path.join(caching_dir, 'mels')
	return path

def _get_wav_dir(caching_dir: str) => str:
	''' the directory to write the preprocessed wav into '''
	path = os.path.join(caching_dir, 'audio')
	return path
	
def _get_linear_spectrograms_dir(caching_dir: str) => str:
	''' the directory to write the linear spectrograms into '''
	path = os.path.join(caching_dir, 'linear')
	return path
	
def _ensure_folders_exist(caching_dir: str):
	os.makedirs(caching_dir)

	mel_dir = _get_mel_dir(caching_dir)
	os.makedirs(_get_mel_dir(caching_dir), exist_ok=True)

	wav_dir = _get_wav_dir(caching_dir)
	os.makedirs(_get_mel_dir(wav_dir), exist_ok=True)

	linear_dir = _get_linear_spectrograms_dir(caching_dir)
	os.makedirs(_get_mel_dir(linear_dir), exist_ok=True)

def main():
	print('initializing preprocessing..')
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset_path', default='/datasets/LJSpeech-1.1-lite')
	parser.add_argument('--cache_path', default='/datasets/models/tacotron/cache')
	parser.add_argument('--n_jobs', type=int, default=cpu_count())



	parser.add_argument('--base_dir', default='')
	parser.add_argument('--hparams', default='',
		help='Hyperparameter overrides as a comma-separated list of name=value pairs')
	parser.add_argument('--dataset', default='LJSpeech-1.1')
	
	parser.add_argument('--language', default='en_US')
	parser.add_argument('--voice', default='female')
	parser.add_argument('--reader', default='mary_ann')
	parser.add_argument('--merge_books', default='False')
	parser.add_argument('--book', default='northandsouth')
	parser.add_argument('--output', default='/datasets/models/tacotron2/training_data')
	args = parser.parse_args()

	if not os.path.exists(args.dataset_path):
		print("Dataset not found", args.dataset_path)

	_ensure_folders_exist(args.cache_path)


	modified_hp = hparams.parse(args.hparams)

	assert args.merge_books in ('False', 'True')

	run_preprocess(args, modified_hp)


if __name__ == '__main__':
	main()
