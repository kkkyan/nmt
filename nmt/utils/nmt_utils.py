# Copyright 2017 Google Inc. All Rights Reserved.
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
# ==============================================================================

"""Utility functions specifically for NMT."""
from __future__ import print_function

import codecs
import time
import numpy as np
import tensorflow as tf

from ..utils import evaluation_utils
from ..utils import misc_utils as utils

__all__ = ["decode_and_evaluate", "get_translation"]


def decode_and_evaluate(name,
                        model,
                        sess,
                        trans_file,
                        ref_file,
                        metrics,
                        subword_option,
                        beam_width,
                        tgt_eos,
                        num_translations_per_input=1,
                        decode=True):
  """Decode a test set and compute a score according to the evaluation task."""
  fw_trans_file, bw_trans_file = trans_file

  # Decode
  if decode:
    utils.print_out("  decoding to output %s." % fw_trans_file)
    utils.print_out("  decoding to output %s." % bw_trans_file)

    start_time = time.time()
    num_sentences = 0
    with codecs.getwriter("utf-8")(
        tf.gfile.GFile(fw_trans_file, mode="wb")) as fw_trans_f,\
        codecs.getwriter("utf-8")(
          tf.gfile.GFile(bw_trans_file, mode="wb")) as bw_trans_f:
      fw_trans_f.write("")  # Write empty string to ensure file is created.
      bw_trans_f.write("")  # Write empty string to ensure file is created.

      num_translations_per_input = max(
          min(num_translations_per_input, beam_width), 1)
      while True:
        try:
          (fw_nmt_outputs, bw_nmt_outputs), _ = model.decode(sess)
          if beam_width == 0:
            fw_nmt_outputs = np.expand_dims(fw_nmt_outputs, 0)
            bw_nmt_outputs = np.expand_dims(bw_nmt_outputs, 0)

          batch_size = fw_nmt_outputs.shape[1]
          num_sentences += batch_size

          for sent_id in range(batch_size):
            for beam_id in range(num_translations_per_input):
              (fw_translation, bw_translation) = get_translation(
                  (fw_nmt_outputs[beam_id], bw_nmt_outputs[beam_id]),
                  sent_id,
                  tgt_eos=tgt_eos,
                  subword_option=subword_option)
              fw_trans_f.write((fw_translation + b"\n").decode("utf-8"))
              bw_trans_f.write((bw_translation + b"\n").decode("utf-8"))
        except tf.errors.OutOfRangeError:
          utils.print_time(
              "  done, num sentences %d, num translations per input %d" %
              (num_sentences, num_translations_per_input), start_time)
          break

  # Evaluation
  evaluation_scores = {}
  if ref_file and (tf.gfile.Exists(fw_trans_file) and tf.gfile.Exists(bw_trans_file)):
    for metric in metrics:
      score = evaluation_utils.evaluate(
          ref_file,
          trans_file,
          metric,
          subword_option=subword_option)
      evaluation_scores[metric] = score
      
      utils.print_out("fw_%s %s: %.1f" % (metric, name, score[0]))
      utils.print_out("bw_%s %s: %.1f" % (metric, name, score[1]))

  return evaluation_scores


def get_translation(nmt_outputs, sent_id, tgt_eos, subword_option):
  """Given batch decoding outputs, select a sentence and turn to text."""
  if tgt_eos: tgt_eos = tgt_eos.encode("utf-8")
  # for 2 direction
  fw_nmt_output, bw_nmt_output = nmt_outputs
  # Select a sentence
  fw_output = fw_nmt_output[sent_id, :].tolist()
  bw_output = bw_nmt_output[sent_id, :].tolist()

  # If there is an eos symbol in outputs, cut them at that point.
  if tgt_eos and tgt_eos in fw_output:
    fw_output = fw_output[:fw_output.index(tgt_eos)]
  if tgt_eos and tgt_eos in bw_output:
    bw_output = bw_output[:bw_output.index(tgt_eos)]

  # backward reverse list
  bw_output.reverse()

  if subword_option == "bpe":  # BPE
    fw_translation = utils.format_bpe_text(fw_output)
    bw_translation = utils.format_bpe_text(bw_output)
  elif subword_option == "spm":  # SPM
    fw_translation = utils.format_spm_text(fw_output)
    bw_translation = utils.format_spm_text(bw_output)
  else:
    fw_translation = utils.format_text(fw_output)
    bw_translation = utils.format_text(bw_output)

  return (fw_translation, bw_translation)
