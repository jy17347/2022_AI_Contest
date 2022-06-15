# Copyright (c) 2020, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from preprocess.preprocess import preprocess
from preprocess.character import generate_character_labels, generate_character_script


def _get_parser():
    """ Get arguments parser """
    parser = argparse.ArgumentParser(description='KsponSpeech Preprocess')
    parser.add_argument('--dataset_path', type=str,
                        default='./data/transcripts/train.txt',
                        help='path of original dataset')
    parser.add_argument('--vocab_dest', type=str,
                        default='./data/vocab',
                        help='destination to save character / subword labels file')
    parser.add_argument('--savepath', type=str,
                        default='./data/transcripts',
                        help='path of data')

    return parser


def log_info(opt):
    print("Dataset Path : %s" % opt.dataset_path)
    print("Vocab Destination : %s" % opt.vocab_dest)
    print("Save Path : %s" % opt.savepath)


def main():
    parser = _get_parser()
    opt = parser.parse_args()
    log_info(opt)

    audio_paths, transcripts = preprocess(opt.dataset_path)
    generate_character_labels(transcripts, opt.vocab_dest)
    generate_character_script(audio_paths, transcripts, opt.vocab_dest, opt.savepath)

if __name__ == '__main__':
    main()
